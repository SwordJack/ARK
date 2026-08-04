#! python3
# -*- encoding: utf-8 -*-
"""Dense vector retrieval implementation.

@File   :   dense.py
@Created:   2026/08/03 19:55 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from typing import List

from ..entities import RetrievalResult
from .base import BaseRetriever, DenseRetrievalItem


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Computes cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity score in range [-1, 1]. Returns 0.0 if either
        vector has zero magnitude.

    Raises:
        ValueError: If vectors have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vectors must have the same length: {len(a)} != {len(b)}"
        )

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class DenseRetriever(BaseRetriever):
    """Ranks chunks by cosine similarity against a query embedding."""

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
            Retrieval results sorted by descending cosine similarity.
        """
        results = [
            RetrievalResult(
                chunk=item.chunk,
                score=cosine_similarity(query_embedding, item.embedding),
                document=item.document,
                score_source="dense",
            )
            for item in items
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
