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
            dimensions=1024
        )

    Example usage with OpenAI:
        client = OpenAIEmbeddingClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-small"
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
            **kwargs: Additional default parameters passed to all requests.

        Raises:
            ValueError: If api_key is empty.
        """
        if not api_key:
            raise ValueError("api_key cannot be empty")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self._dimensions = dimensions
        self.default_kwargs = kwargs

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
            return [item.embedding for item in response.data]
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

    @property
    def dimensions(self) -> int:
        """Returns embedding vector dimensions.

        Returns:
            Dimension size. If not explicitly set, infers from model name.

        Note:
            - text-embedding-3-small: 1536
            - text-embedding-3-large: 3072
            - text-embedding-v4 (Qwen): 1024 (default)
            - text-embedding-v3 (Qwen): 1024 (default)
        """
        if self._dimensions:
            return self._dimensions

        # Infer from model name
        model_lower = self.model.lower()

        if "text-embedding-3-large" in model_lower:
            return 3072
        elif "text-embedding-3-small" in model_lower:
            return 1536
        elif "text-embedding-v4" in model_lower:
            return 1024
        elif "text-embedding-v3" in model_lower:
            return 1024

        # Conservative default
        return 1536
