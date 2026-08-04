#! python3
# -*- encoding: utf-8 -*-
"""Retrieval pipeline — the single entry point for retrieval logic.

@File   :   pipeline.py
@Created:   2026/08/04 21:47 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from typing import Dict, List, Optional

from ..entities import RetrievalOption, RetrievalResult
from ..stores.base import BaseKnowledgeStore
from .reranker import BaseReranker, NoOpReranker
from .sparse import SparseRetriever, SparseRetrievalItem


class RetrievalPipeline:
    """Orchestrates retrieval across the store plus options.

    The pipeline owns the flow: gather candidates from the store,
    apply metadata filtering client-side, and trim to final top_k.

    Store implementations do not need to know about metadata filters
    or candidate/final_k — the pipeline handles all of that.
    """

    def __init__(
        self,
        store: BaseKnowledgeStore,
        reranker: Optional[BaseReranker] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
    ) -> None:
        """Creates a pipeline for the given store.

        Args:
            store: Knowledge store backend.
            reranker: Optional post-retrieval reranker. Defaults to
                ``NoOpReranker`` to preserve dense-only behavior.
            sparse_retriever: Optional sparse retriever used when
                ``RetrievalOption.use_sparse`` is enabled.
        """
        self._store = store
        self._reranker = reranker or NoOpReranker()
        self._sparse_retriever = sparse_retriever or SparseRetriever()

    def search(
        self,
        kb_id: str,
        query_embedding: List[float],
        options: RetrievalOption,
        query_text: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Retrieves and ranks results for the given query.

        Args:
            kb_id: Knowledge base ID to search within.
            query_embedding: Query vector.
            options: Retrieval options controlling candidate_k, top_k,
                and metadata_filter.
            query_text: Original query text, forwarded to the reranker
                when available (required by model-based rerankers).

        Returns:
            Up to ``options.top_k`` ranked results, sorted by score descending.
            Empty list if the knowledge base is empty.
        """
        candidate_k = options.effective_candidate_k()

        if options.use_sparse:
            if query_text is None:
                raise ValueError("query_text is required when use_sparse=True")
            results = self._search_sparse(kb_id, query_text, candidate_k)
        else:
            results = self._store.search(kb_id, query_embedding, top_k=candidate_k)

        if options.metadata_filter:
            results = _apply_metadata_filter(results, options.metadata_filter)

        reranked = self._reranker.rerank(query_text, results, options.top_k)
        return reranked[: options.top_k]

    def _search_sparse(
        self,
        kb_id: str,
        query_text: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        """Runs sparse retrieval over stored chunks.

        Args:
            kb_id: Knowledge base ID to search within.
            query_text: Query text.
            top_k: Maximum number of results to return.

        Returns:
            Sparse retrieval results sorted by descending BM25 score.
        """
        items = [
            SparseRetrievalItem(chunk=chunk, document=self._store.get_document(chunk.document_id))
            for chunk in self._store.list_chunks(kb_id)
        ]
        return self._sparse_retriever.retrieve(query_text, items, top_k)


def _apply_metadata_filter(
    results: List[RetrievalResult],
    filters: Dict[str, object],
) -> List[RetrievalResult]:
    """Filters results whose chunk metadata must contain all key/value pairs.

    Args:
        results: Ranked retrieval results.
        filters: Metadata equality filters.

    Returns:
        Results whose chunk metadata contains every filter key/value pair.
    """
    return [
        r
        for r in results
        if all(
            r.chunk.metadata.get(k) == v for k, v in filters.items()
        )
    ]
