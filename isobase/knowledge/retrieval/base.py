#! python3
# -*- encoding: utf-8 -*-
"""Abstract base classes for knowledge retrieval strategies.

@File   :   base.py
@Created:   2026/08/03 19:55 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from ..entities import KnowledgeChunk, KnowledgeDocument, RetrievalResult


@dataclass
class DenseRetrievalItem:
    """A stored chunk and vector pair available for dense retrieval.

    Args:
        chunk: Indexed chunk to score.
        embedding: Embedding vector for the chunk.
        document: Optional parent document for result provenance.
    """

    chunk: KnowledgeChunk
    embedding: List[float]
    document: Optional[KnowledgeDocument] = None


class BaseRetriever(ABC):
    """Abstract retrieval strategy interface."""

    @abstractmethod
    def retrieve(
        self,
        query_embedding: List[float],
        items: List[DenseRetrievalItem],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Ranks candidate chunks for a query embedding.

        Args:
            query_embedding: Query vector.
            items: Candidate chunk/vector pairs.
            top_k: Maximum number of results to return.

        Returns:
            Retrieval results sorted by descending relevance score.
        """
        pass
