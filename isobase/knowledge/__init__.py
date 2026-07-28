#! python3
# -*- encoding: utf-8 -*-
"""Knowledge base and RAG (Retrieval-Augmented Generation) capabilities.

This module provides provider-neutral knowledge base functionality for
indexing documents and retrieving relevant context for LLM applications.

Key components:
- entities: Core data structures (KnowledgeBase, Document, Chunk, RetrievalResult)
- embeddings: Text embedding clients (OpenAI-compatible)
- chunking: Document splitting strategies
- stores: Storage backends (memory, SQL)
- service: High-level orchestration (KnowledgeBaseService)
- tools: LLM tool integration (FunctionTool wrappers)

@File   :   __init__.py
@Created:   2026/07/29 00:16
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from .entities import (
    KnowledgeBase,
    KnowledgeDocument,
    KnowledgeChunk,
    RetrievalResult,
)
from .service import KnowledgeBaseService

__all__ = [
    "KnowledgeBase",
    "KnowledgeDocument",
    "KnowledgeChunk",
    "RetrievalResult",
    "KnowledgeBaseService",
]
