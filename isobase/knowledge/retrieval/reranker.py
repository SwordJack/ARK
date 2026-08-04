#! python3
# -*- encoding: utf-8 -*-
"""Reranker hooks for knowledge retrieval pipelines.

@File   :   reranker.py
@Created:   2026/08/04 22:20 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..entities import RetrievalResult


class BaseReranker(ABC):
    """Abstract reranker interface for post-retrieval ranking."""

    @abstractmethod
    def rerank(
        self,
        query: Optional[str],
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Reranks candidate retrieval results.

        Args:
            query: Original query text when available.
            results: Candidate retrieval results from the first-stage retriever.
            top_k: Maximum number of final results to return.

        Returns:
            Reranked retrieval results.
        """
        pass


class NoOpReranker(BaseReranker):
    """Default reranker that preserves existing dense-only behavior."""

    def rerank(
        self,
        query: Optional[str],
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Returns the input ranking unchanged, trimmed to ``top_k``.

        Args:
            query: Original query text when available. Unused.
            results: Candidate retrieval results.
            top_k: Maximum number of final results to return.

        Returns:
            The original results order, limited to ``top_k``.
        """
        return results[:top_k]
