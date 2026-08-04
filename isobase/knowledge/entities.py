#! python3
# -*- encoding: utf-8 -*-
"""Core data structures for knowledge base operations.

@File   :   entities.py
@Created:   2026/07/29 00:16
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class RetrievalOption:
    """Options for retrieval candidate selection and filtering.

    Attributes:
        top_k: Final number of results to return.
        candidate_k: Number of candidates to collect before final trimming.
            When None, defaults to top_k.
        metadata_filter: Chunk metadata equality filters. Only chunks whose
            metadata contains all key/value pairs are retained.
        use_sparse: Whether to use sparse text retrieval instead of dense
            vector retrieval. Defaults to False to preserve dense-only behavior.
    """

    top_k: int = 5
    candidate_k: Optional[int] = None
    metadata_filter: Dict[str, Any] = field(default_factory=dict)
    use_sparse: bool = False

    def effective_candidate_k(self) -> int:
        """Returns the candidate count to use for the initial retrieval pass."""
        if self.candidate_k is None:
            return self.top_k
        return max(self.candidate_k, self.top_k)


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

    id: str = ""
    name: str = ""
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

    id: str = ""
    knowledge_base_id: str = ""
    title: str = ""
    source_uri: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_time: Optional[datetime] = None


@dataclass
class KnowledgeChunk:
    """Represents an indexed text chunk.

    Chunks are the unit of retrieval. Each chunk is embedded and stored
    with its vector representation for similarity search.

    ``metadata`` is a free-form dictionary.  To make the most of future
    semantic chunkers, consider using these (all optional) keys:

    * ``heading_path``: ``List[str]`` — breadcrumb trail of parent headings,
      e.g. ``["第 1 章", "1.1 概述"]``.
    * ``chunk_strategy``: ``str`` — name of the strategy that produced this
      chunk (``"fixed"``, ``"recursive"``, ``"markdown"``, …).

    Attributes:
        id: Unique identifier for the chunk.
        document_id: Parent document ID.
        knowledge_base_id: Parent knowledge base ID.
        content: Text content of the chunk.
        index: Position of this chunk within the parent document (0-based).
        token_count: Approximate token count (for cost estimation).
        metadata: Additional metadata (heading path, page number, etc.).
    """

    id: str = ""
    document_id: str = ""
    knowledge_base_id: str = ""
    content: str = ""
    index: int = 0
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
        score_source: Source of score semantics, e.g. ``"dense"``.
    """

    chunk: KnowledgeChunk
    score: float
    document: Optional[KnowledgeDocument] = None
    score_source: str = "dense"

    def __str__(self) -> str:
        """Formats the result for LLM context injection.

        Returns a human-readable representation with source attribution
        suitable for including in LLM prompts.
        """
        source = f" ({self.document.title})" if self.document else ""
        return f"[Score: {self.score:.3f}{source}]\n{self.chunk.content}"
