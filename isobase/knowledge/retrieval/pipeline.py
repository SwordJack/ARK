#! python3
# -*- encoding: utf-8 -*-
"""Retrieval pipeline — the single entry point for retrieval logic.

@File   :   pipeline.py
@Created:   2026/08/04 21:47 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from typing import Dict, List

from ..entities import RetrievalOption, RetrievalResult
from ..stores.base import BaseKnowledgeStore


class RetrievalPipeline:
    """Orchestrates retrieval across the store plus options.

    The pipeline owns the flow: gather candidates from the store,
    apply metadata filtering client-side, and trim to final top_k.

    Store implementations do not need to know about metadata filters
    or candidate/final_k — the pipeline handles all of that.
    """

    def __init__(self, store: BaseKnowledgeStore) -> None:
        """Creates a pipeline for the given store.

        Args:
            store: Knowledge store backend.
        """
        self._store = store

    def search(
        self,
        kb_id: str,
        query_embedding: List[float],
        options: RetrievalOption,
    ) -> List[RetrievalResult]:
        """Retrieves and ranks results for the given query.

        Args:
            kb_id: Knowledge base ID to search within.
            query_embedding: Query vector.
            options: Retrieval options controlling candidate_k, top_k,
                and metadata_filter.

        Returns:
            Up to ``options.top_k`` ranked results, sorted by score descending.
            Empty list if the knowledge base is empty.
        """
        candidate_k = options.effective_candidate_k()

        results = self._store.search(kb_id, query_embedding, top_k=candidate_k)

        if options.metadata_filter:
            results = _apply_metadata_filter(results, options.metadata_filter)

        return results[: options.top_k]


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
