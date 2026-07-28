#! python3
# -*- encoding: utf-8 -*-
"""Abstract base class for text embedding clients.

@File   :   base.py
@Created:   2026/07/29 00:20
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingClient(ABC):
    """Abstract embedding client interface.

    Separated from BaseLLMClient because embedding is a different task
    with different input/output contracts, pricing models, and rate limits.

    Embedding models convert text into fixed-dimensional vectors that
    capture semantic meaning for similarity search.
    """

    @abstractmethod
    def embed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Embeds a batch of texts.

        Args:
            texts: List of text strings to embed.
            **kwargs: Model-specific parameters (e.g., dimensions, encoding_format).

        Returns:
            List of embedding vectors (each vector is a list of floats).
            The order matches the input texts order.

        Raises:
            ValueError: If texts is empty or exceeds batch size limits.
            RuntimeError: If API call fails after retries.

        Example:
            >>> client = SomeEmbeddingClient(...)
            >>> vectors = client.embed_texts(["hello", "world"])
            >>> len(vectors)
            2
            >>> len(vectors[0])  # dimension size
            1024
        """
        pass

    @abstractmethod
    def embed_query(self, query: str, **kwargs) -> List[float]:
        """Embeds a single query string.

        Some embedding models distinguish between document embeddings
        (for indexing) and query embeddings (for retrieval). Default
        implementations may delegate to embed_texts()[0].

        Args:
            query: Query text to embed.
            **kwargs: Model-specific parameters.

        Returns:
            Embedding vector (list of floats).

        Raises:
            ValueError: If query is empty.
            RuntimeError: If API call fails.

        Example:
            >>> client = SomeEmbeddingClient(...)
            >>> vector = client.embed_query("search term")
            >>> len(vector)
            1024
        """
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Returns the embedding vector dimensions.

        This value must remain constant for a given model configuration.
        All vectors returned by embed_texts() and embed_query() must
        have this length.

        Returns:
            Dimension size (e.g., 1024, 1536, 3072).
        """
        pass
