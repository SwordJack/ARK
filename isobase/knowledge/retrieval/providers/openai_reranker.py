#! python3
# -*- encoding: utf-8 -*-
"""Provider-specific reranker implementations.

@File   :   openai_reranker.py
@Created:   2026/08/04 22:43 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from typing import List, Optional

from openai import OpenAI

from ..reranker import BaseReranker
from ...entities import RetrievalResult


class OpenAIReranker(BaseReranker):
    """Calls an OpenAI-compatible ``/reranks`` endpoint for scoring.

    Supports any rerank API that follows the OpenAI-compatible rerank
    specification, including:

    - Alibaba DashScope (qwen3-rerank via MaaS compatible-api)
    - Any custom OpenAI-compatible rerank endpoint

    .. note::

        The DashScope ``dashscope.aliyuncs.com`` domain does **not** host a
        ``/reranks`` endpoint.  You must use a MaaS workspace URL::

            https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-api/v1

    Example usage::

        from isobase.knowledge.retrieval.providers import OpenAIReranker

        reranker = OpenAIReranker(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://{workspace}.cn-beijing.maas.aliyuncs.com/"
                     "compatible-api/v1",
            model="qwen3-rerank",
        )
        pipeline = RetrievalPipeline(store, reranker=reranker)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "qwen3-rerank",
        endpoint: str = "/reranks",
    ) -> None:
        """Initializes the OpenAI-compatible reranker.

        Args:
            api_key: API key for authentication.
            base_url: Base URL for the rerank endpoint.
            model: Model identifier (default: ``"qwen3-rerank"``).
            endpoint: Rerank endpoint path (default: ``"/reranks"``).
                Use ``"/rerank"`` for providers that deviate from the
                ``/reranks`` convention.

        Raises:
            ValueError: If api_key is empty.
        """
        if not api_key:
            raise ValueError("api_key cannot be empty")

        self.model = model
        self.endpoint = endpoint
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def rerank(
        self,
        query: Optional[str],
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Reranks candidate retrieval results via the ``/reranks`` endpoint.

        Args:
            query: Original query text.
            results: Candidate retrieval results from the first-stage retriever.
            top_k: Maximum number of final results to return.

        Returns:
            Reranked results with scores from the model and
            ``score_source`` set to ``"rerank"``.
        """
        if not results:
            return []

        # Build document list
        documents = [r.chunk.content for r in results]

        response = self.client.post(
            self.endpoint,
            body={
                "model": self.model,
                "query": query or "",
                "documents": documents,
                "top_n": top_k,
            },
            cast_to=object,
        )

        # Parse response — cast_to=object returns dict on some SDK versions
        results_list = (
            response.get("results") if isinstance(response, dict)
            else getattr(response, "results", None)
        )
        if not results_list:
            return []

        # Build reranked output
        reranked: List[RetrievalResult] = []
        for item in results_list:
            idx = (
                item.get("index") if isinstance(item, dict)
                else getattr(item, "index", None)
            )
            relevance = (
                item.get("relevance_score") if isinstance(item, dict)
                else getattr(item, "relevance_score", None)
            )
            if idx is None or relevance is None:
                continue
            original = results[idx]
            reranked.append(
                RetrievalResult(
                    chunk=original.chunk,
                    score=relevance,
                    document=original.document,
                    score_source="rerank",
                )
            )

        return reranked[:top_k]
