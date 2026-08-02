#! python3
# -*- encoding: utf-8 -*-
"""Storage backends for knowledge bases and vectors.

Store Contract
--------------

Every backend in this package implements :class:`BaseKnowledgeStore`.  They
share a common behaviour contract verified by
``test/knowledge/stores/test_store_contract.py``.

When implementing a new store, observe these constraints:

- The store persists an *indexing snapshot*, not a live mirror of the
  original file.  ``source_uri`` records provenance only; the store must
  not assume that the external resource still exists or is accessible.
- ``KnowledgeChunk.content`` is the direct source of retrieval context.
  It must survive a round-trip through ``add_chunks()`` → ``search()``.
- ``KnowledgeDocument.content`` is an optional document-text snapshot.
  It is not the single source of truth for any external file.
- Cascade semantics: ``delete_document()`` removes the document together
  with every chunk and embedding that belongs to it; ``delete_knowledge_base()``
  additionally removes all documents.
- Error conventions:
  * ``ValueError`` — duplicate KB id, chunk/embedding count mismatch,
    vector dimension mismatch.
  * ``KeyError`` — missing KB, missing document (including list/delete
    operations on a non-existent parent).

@File   :   __init__.py
@Created:   2026/07/29 00:20
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from .base import BaseKnowledgeStore
from .memory import MemoryKnowledgeStore
from .mongo import MongoKnowledgeStore
from .sql import SqlKnowledgeStore

__all__ = [
    "BaseKnowledgeStore",
    "MemoryKnowledgeStore",
    "MongoKnowledgeStore",
    "SqlKnowledgeStore",
]
