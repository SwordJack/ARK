#! python3
# -*- encoding: utf-8 -*-
"""OpenAI-compatible embedding client implementation.

@File   :   openai.py
@Created:   2026/07/29 00:20
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from typing import List, Optional
from openai import OpenAI
from .base import BaseEmbeddingClient


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """OpenAI-compatible embedding client.

    Supports any embedding API that follows the OpenAI embeddings endpoint
    specification, including:

    - OpenAI (text-embedding-3-small, text-embedding-3-large)
    - Alibaba DashScope (text-embedding-v3, text-embedding-v4)
    - AIMLAPI (alibaba/qwen-text-embedding-v4)
    - Azure OpenAI
    - Any custom OpenAI-compatible endpoint

    Dimensions are resolved in two ways:

        1. Explicit ``dimensions`` parameter (highest priority).
        2. Auto-detected from the first API response.

    If neither is available when ``.dimensions`` is accessed, a
    ``RuntimeError`` is raised.  Use ``fetch_dimensions()`` to actively
    discover the dimension count before embedding real content.

    Example usage with Qwen v4 via AIMLAPI:
        client = OpenAIEmbeddingClient(
            api_key="your_aimlapi_key",
            base_url="https://api.aimlapi.com/v1",
            model="alibaba/qwen-text-embedding-v4",
            dimensions=1024
        )

    Example usage with Alibaba DashScope:
        import os
        client = OpenAIEmbeddingClient(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="text-embedding-v4",
        )
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        **kwargs
    ):
        """Initializes the OpenAI-compatible embedding client.

        Args:
            api_key: API key for authentication.
            base_url: Base URL for the embeddings endpoint.
            model: Model identifier (e.g., "text-embedding-3-small").
            dimensions: Optional dimension override. Some models support
                configurable dimensions (e.g., Qwen v4: 512/1024/2048).
                If not provided, will be auto-detected from the first
                API response.
            **kwargs: Additional default parameters passed to all requests.

        Raises:
            ValueError: If api_key is empty.
        """
        if not api_key:
            raise ValueError("api_key cannot be empty")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self._dimensions = dimensions
        self._inferred_dimensions: Optional[int] = None
        self.default_kwargs = kwargs

    def _infer_dimensions_from_response(self, embedding: List[float]) -> None:
        """Cache the dimension count from a single embedding vector."""
        if not self._inferred_dimensions:
            self._inferred_dimensions = len(embedding)

    # ------------------------------------------------------------------
    # BaseEmbeddingClient implementation
    # ------------------------------------------------------------------

    def embed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Embeds a batch of texts.

        Args:
            texts: List of text strings to embed.
            **kwargs: Additional parameters for this request (overrides defaults).

        Returns:
            List of embedding vectors.

        Raises:
            ValueError: If texts is empty.
            RuntimeError: If API call fails.

        Note:
            - OpenAI allows up to ~2048 texts per request
            - DashScope allows up to 25 texts per request
            - Consider batching for large document sets
        """
        if not texts:
            raise ValueError("texts cannot be empty")

        params = {**self.default_kwargs, **kwargs}
        if self._dimensions:
            params["dimensions"] = self._dimensions

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                **params
            )

            embeddings = [item.embedding for item in response.data]
            if embeddings:
                self._infer_dimensions_from_response(embeddings[0])

            return embeddings
        except Exception as e:
            raise RuntimeError(f"Embedding API call failed: {e}") from e

    def embed_query(self, query: str, **kwargs) -> List[float]:
        """Embeds a single query string.

        Args:
            query: Query text to embed.
            **kwargs: Additional parameters for this request.

        Returns:
            Embedding vector.

        Raises:
            ValueError: If query is empty.
            RuntimeError: If API call fails.
        """
        if not query:
            raise ValueError("query cannot be empty")

        return self.embed_texts([query], **kwargs)[0]

    # ------------------------------------------------------------------
    # Dimension helpers
    # ------------------------------------------------------------------

    @property
    def dimensions(self) -> int:
        """Returns embedding vector dimensions.

        Resolution order:

            1. Explicit ``dimensions`` set in constructor.
            2. Auto-detected from the most recent API response.

        Returns:
            Dimension size.

        Raises:
            RuntimeError: If dimensions have not been set explicitly
                and no API call has been made yet.

        Example:
            >>> # Explicit setting
            >>> client = OpenAIEmbeddingClient(api_key="...", dimensions=1024)
            >>> client.dimensions
            1024

            >>> # Auto-detect from response
            >>> client = OpenAIEmbeddingClient(api_key="...")
            >>> client.embed_texts(["test"])
            >>> client.dimensions  # detected from response
            1536
        """
        if self._dimensions:
            return self._dimensions

        if self._inferred_dimensions:
            return self._inferred_dimensions

        raise RuntimeError(
            "Dimensions unknown. "
            "Call embed_texts() or embed_query() first to auto-detect, "
            "or pass dimensions explicitly: OpenAIEmbeddingClient(..., dimensions=N)"
        )

    def fetch_dimensions(self) -> int:
        """Send a lightweight request to determine the model's output dimensions.

        This makes exactly one API call with a minimal input (``"test"``),
        stores the result, and returns the dimension count.  Subsequent
        calls (or accessing the ``.dimensions`` property) will use the
        cached value and **not** issue another request.

        Use this when you want to know the dimensions upfront without
        embedding real content.

        Returns:
            Embedding vector dimension count.

        Raises:
            RuntimeError: If the API call fails.

        Example:
            >>> client = OpenAIEmbeddingClient(api_key="...", model="text-embedding-v4")
            >>> dims = client.fetch_dimensions()
            >>> dims
            1024
            >>> client.dimensions  # uses cached value
            1024
        """
        if self._dimensions:
            return self._dimensions

        if self._inferred_dimensions:
            return self._inferred_dimensions

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=["test"],
            )

            if response.data:
                embedding = response.data[0].embedding
                self._inferred_dimensions = len(embedding)
                return self._inferred_dimensions

            raise RuntimeError("Empty response from embedding API")
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch dimensions for model '{self.model}': {e}"
            ) from e
